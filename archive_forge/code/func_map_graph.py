import collections
import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.config import backend
from keras.src.ops.operation import Operation
from keras.src.utils.nest import pack_sequence_as
def map_graph(inputs, outputs):
    """Validates a graph's topology and gather its operations and nodes.

    Args:
        inputs: List of input tensors.
        outputs: List of outputs tensors.

    Returns:
        A tuple `(nodes, nodes_by_depth, operations, operations_by_depth)`.
        - network_nodes: dict mapping unique node keys to the Node instances
        - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
        - operations: list of Operation instances.
        - operations_by_depth: dict mapping ints (depth) to lists of Operation
            instances.
    """
    nodes_in_decreasing_depth, operation_indices = _build_map(outputs)
    network_nodes = {make_node_key(node.operation, node.operation._inbound_nodes.index(node)) for node in nodes_in_decreasing_depth}
    nodes_depths = {}
    operations_depths = {}
    for node in reversed(nodes_in_decreasing_depth):
        depth = nodes_depths.setdefault(node, 0)
        previous_depth = operations_depths.get(node.operation, 0)
        depth = max(depth, previous_depth)
        operations_depths[node.operation] = depth
        nodes_depths[node] = depth
        for node_dep in node.parent_nodes:
            previous_depth = nodes_depths.get(node_dep, 0)
            nodes_depths[node_dep] = max(depth + 1, previous_depth)
    for input_t in inputs:
        input_operation = input_t._keras_history[0]
        if input_operation and input_operation not in operations_depths:
            operations_depths[input_operation] = 0
            operation_indices[input_operation] = -1
            nodes_depths[input_operation._inbound_nodes[0]] = 0
            network_nodes.add(make_node_key(input_operation, 0))
    nodes_by_depth = collections.defaultdict(list)
    for node, depth in nodes_depths.items():
        nodes_by_depth[depth].append(node)
    operations_by_depth = collections.defaultdict(list)
    for operation, depth in operations_depths.items():
        operations_by_depth[depth].append(operation)
    depth_keys = list(operations_by_depth.keys())
    depth_keys.sort(reverse=True)
    operations = []
    for depth in depth_keys:
        operations_for_depth = operations_by_depth[depth]
        operations_for_depth.sort(key=lambda x: operation_indices[x])
        operations.extend(operations_for_depth)
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    computable_tensors = set()
    for x in inputs:
        computable_tensors.add(x)
    operations_with_complete_input = []
    for depth in depth_keys:
        for node in nodes_by_depth[depth]:
            for x in tree.flatten(node.input_tensors):
                if x not in computable_tensors:
                    operation = node.operation
                    raise ValueError(f"Graph disconnected: cannot find parent for tensor {x} at operation '{operation}'. The following previous operations were accessed without issue: {operations_with_complete_input}")
                operations_with_complete_input.append(operation.name)
            for x in tree.flatten(node.outputs):
                computable_tensors.add(x)
    all_names = [operation.name for operation in operations]
    for name in all_names:
        if all_names.count(name) != 1:
            raise ValueError(f'The name "{name}" is used {all_names.count(name)} times in the model. All operation names should be unique.')
    return (network_nodes, nodes_by_depth, operations, operations_by_depth)