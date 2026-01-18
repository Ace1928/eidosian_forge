from pathlib import Path
from typing import TYPE_CHECKING
from .. import BaseOptimumCLICommand
def parse_args_onnxruntime_quantize(parser: 'ArgumentParser'):
    required_group = parser.add_argument_group('Required arguments')
    required_group.add_argument('--onnx_model', type=Path, required=True, help='Path to the repository where the ONNX models to quantize are located.')
    required_group.add_argument('-o', '--output', type=Path, required=True, help='Path to the directory where to store generated ONNX model.')
    optional_group = parser.add_argument_group('Optional arguments')
    optional_group.add_argument('--per_channel', action='store_true', help='Compute the quantization parameters on a per-channel basis.')
    level_group = parser.add_mutually_exclusive_group(required=True)
    level_group.add_argument('--arm64', action='store_true', help='Quantization for the ARM64 architecture.')
    level_group.add_argument('--avx2', action='store_true', help='Quantization with AVX-2 instructions.')
    level_group.add_argument('--avx512', action='store_true', help='Quantization with AVX-512 instructions.')
    level_group.add_argument('--avx512_vnni', action='store_true', help='Quantization with AVX-512 and VNNI instructions.')
    level_group.add_argument('--tensorrt', action='store_true', help='Quantization for NVIDIA TensorRT optimizer.')
    level_group.add_argument('-c', '--config', type=Path, help='`ORTConfig` file to use to optimize the model.')