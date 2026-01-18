from pathlib import Path
from typing import TYPE_CHECKING
from optimum.commands.base import BaseOptimumCLICommand
def parse_args_onnxruntime_optimize(parser: 'ArgumentParser'):
    required_group = parser.add_argument_group('Required arguments')
    required_group.add_argument('--onnx_model', type=Path, required=True, help='Path to the repository where the ONNX models to optimize are located.')
    required_group.add_argument('-o', '--output', type=Path, required=True, help='Path to the directory where to store generated ONNX model.')
    level_group = parser.add_mutually_exclusive_group(required=True)
    level_group.add_argument('-O1', action='store_true', help='Basic general optimizations (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details).')
    level_group.add_argument('-O2', action='store_true', help='Basic and extended general optimizations, transformers-specific fusions (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details).')
    level_group.add_argument('-O3', action='store_true', help='Same as O2 with Gelu approximation (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details).')
    level_group.add_argument('-O4', action='store_true', help='Same as O3 with mixed precision (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details).')
    level_group.add_argument('-c', '--config', type=Path, help='`ORTConfig` file to use to optimize the model.')