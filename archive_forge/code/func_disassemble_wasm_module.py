from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def disassemble_wasm_module(script_id: runtime.ScriptId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.Optional[str], int, typing.List[int], WasmDisassemblyChunk]]:
    """


    **EXPERIMENTAL**

    :param script_id: Id of the script to disassemble
    :returns: A tuple with the following items:

        0. **streamId** - *(Optional)* For large modules, return a stream from which additional chunks of disassembly can be read successively.
        1. **totalNumberOfLines** - The total number of lines in the disassembly text.
        2. **functionBodyOffsets** - The offsets of all function bodies, in the format [start1, end1, start2, end2, ...] where all ends are exclusive.
        3. **chunk** - The first chunk of disassembly.
    """
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.disassembleWasmModule', 'params': params}
    json = (yield cmd_dict)
    return (str(json['streamId']) if 'streamId' in json else None, int(json['totalNumberOfLines']), [int(i) for i in json['functionBodyOffsets']], WasmDisassemblyChunk.from_json(json['chunk']))