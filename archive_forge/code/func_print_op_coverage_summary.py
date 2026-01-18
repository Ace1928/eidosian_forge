from operator import itemgetter
from functorch.compile import make_boxed_func
import torch
import torch.nn as nn
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table
from torch.distributed._tensor import DTensor
def print_op_coverage_summary(model: nn.Module, args, kwargs, *, output_csv=False):
    """
    Util to print the operator coverage summary of a certain model with tabulute.

    Must have tabulate module installed.
    """
    import csv
    from tabulate import tabulate
    fwd_graph, bwd_graph = get_inductor_decomp_graphs(model, args, kwargs)
    op_counts = {}
    for node in fwd_graph.graph.nodes:
        if node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload):
            if node.target not in op_counts:
                op_counts[node.target] = 0
            op_counts[node.target] += 1
    for node in bwd_graph.graph.nodes:
        if node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload):
            if node.target not in op_counts:
                op_counts[node.target] = 0
            op_counts[node.target] += 1
    op_infos = []
    for op, count in op_counts.items():
        supported = op in DTensor._op_dispatcher.sharding_propagator.op_to_rules
        op_infos.append([op, str(op._schema), count, supported])
    count_idx = 2
    op_infos.sort(key=itemgetter(count_idx), reverse=True)
    headers = ['Operator', 'Schema', 'Total Count', 'Supported']
    print(tabulate(op_infos, headers=headers))
    if output_csv:
        with open('op_summary.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(headers)
            for row in op_infos:
                csv_writer.writerow(row)