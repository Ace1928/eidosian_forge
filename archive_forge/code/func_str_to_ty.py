import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def str_to_ty(name):
    language = tl
    if name[0] == '*':
        ty = str_to_ty(name[1:])
        return language.pointer_type(ty)
    tys = {'fp8e4nv': language.float8e4nv, 'fp8e5': language.float8e5, 'fp8e4b15': language.float8e4b15, 'fp8e4b15x4': language.float8e4b15x4, 'fp16': language.float16, 'bf16': language.bfloat16, 'fp32': language.float32, 'fp64': language.float64, 'i1': language.int1, 'i8': language.int8, 'i16': language.int16, 'i32': language.int32, 'i64': language.int64, 'u8': language.uint8, 'u16': language.uint16, 'u32': language.uint32, 'u64': language.uint64, 'B': language.int1}
    return tys[name]