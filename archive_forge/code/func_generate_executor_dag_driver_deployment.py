import inspect
from collections import OrderedDict
from typing import List
from ray.dag import ClassNode, DAGNode
from ray.dag.function_node import FunctionNode
from ray.dag.utils import _DAGNodeNameGenerator
from ray.experimental.gradio_utils import type_to_string
from ray.serve._private.constants import (
from ray.serve._private.deployment_executor_node import DeploymentExecutorNode
from ray.serve._private.deployment_function_executor_node import (
from ray.serve._private.deployment_function_node import DeploymentFunctionNode
from ray.serve._private.deployment_node import DeploymentNode
from ray.serve.deployment import Deployment, schema_to_deployment
from ray.serve.handle import DeploymentHandle, RayServeHandle
from ray.serve.schema import DeploymentSchema
def generate_executor_dag_driver_deployment(serve_executor_dag_root_node: DAGNode, original_driver_deployment: Deployment):
    """Given a transformed minimal execution serve dag, and original DAGDriver
    deployment, generate new DAGDriver deployment that uses new serve executor
    dag as init_args.

    Args:
        serve_executor_dag_root_node: Transformed
            executor serve dag with only barebone deployment handles.
        original_driver_deployment: User's original DAGDriver
            deployment that wrapped Ray DAG as init args.
    Returns:
        executor_dag_driver_deployment: New DAGDriver deployment
            with executor serve dag as init args.
    """

    def replace_with_handle(node):
        if isinstance(node, DeploymentExecutorNode):
            return node._deployment_handle
        elif isinstance(node, DeploymentFunctionExecutorNode):
            assert len(node.get_args()) == 0 and len(node.get_kwargs()) == 0
            return node._deployment_function_handle
    replaced_deployment_init_args, replaced_deployment_init_kwargs = serve_executor_dag_root_node.apply_functional([serve_executor_dag_root_node.get_args(), serve_executor_dag_root_node.get_kwargs()], predictate_fn=lambda node: isinstance(node, (DeploymentExecutorNode, DeploymentFunctionExecutorNode)), apply_fn=replace_with_handle)
    return original_driver_deployment.options(_init_args=replaced_deployment_init_args, _init_kwargs=replaced_deployment_init_kwargs, _internal=True)