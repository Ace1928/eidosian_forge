from ..cutlass_utils import try_import_cutlass
def instance_template(self):
    return '\n        ${compile_guard_start}\n          using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;\n          manifest.append(\n            new ${gemm_kind}<GemmKernel>("${operation_name}"));\n        ${compile_guard_end}\n        '