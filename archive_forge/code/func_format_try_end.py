from io import StringIO
from typing import List, Union
from bytecode.bytecode import (  # noqa
from bytecode.cfg import BasicBlock, ControlFlowGraph  # noqa
from bytecode.concrete import _ConvertBytecodeToConcrete  # noqa
from bytecode.concrete import ConcreteBytecode, ConcreteInstr
from bytecode.flags import CompilerFlags
from bytecode.instr import (  # noqa
from bytecode.version import __version__
def format_try_end(instr: TryEnd) -> str:
    i = try_begins.index(instr.entry) if instr.entry in try_begins else '<unknwon>'
    return 'TryEnd (%s)' % i