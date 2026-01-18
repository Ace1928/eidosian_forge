import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.concrete import ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.flags import CompilerFlags
from _pydevd_frame_eval.vendored.bytecode.instr import Label, SetLineno, Instr
@staticmethod
def from_bytecode(bytecode):
    label_to_block_index = {}
    jumps = []
    block_starts = {}
    for index, instr in enumerate(bytecode):
        if isinstance(instr, Label):
            label_to_block_index[instr] = index
        elif isinstance(instr, Instr) and isinstance(instr.arg, Label):
            jumps.append((index, instr.arg))
    for target_index, target_label in jumps:
        target_index = label_to_block_index[target_label]
        block_starts[target_index] = target_label
    bytecode_blocks = _bytecode.ControlFlowGraph()
    bytecode_blocks._copy_attr_from(bytecode)
    bytecode_blocks.argnames = list(bytecode.argnames)
    block = bytecode_blocks[0]
    labels = {}
    jumps = []
    for index, instr in enumerate(bytecode):
        if index in block_starts:
            old_label = block_starts[index]
            if index != 0:
                new_block = bytecode_blocks.add_block()
                if not block[-1].is_final():
                    block.next_block = new_block
                block = new_block
            if old_label is not None:
                labels[old_label] = block
        elif block and isinstance(block[-1], Instr):
            if block[-1].is_final():
                block = bytecode_blocks.add_block()
            elif block[-1].has_jump():
                new_block = bytecode_blocks.add_block()
                block.next_block = new_block
                block = new_block
        if isinstance(instr, Label):
            continue
        if isinstance(instr, Instr):
            instr = instr.copy()
            if isinstance(instr.arg, Label):
                jumps.append(instr)
        block.append(instr)
    for instr in jumps:
        label = instr.arg
        instr.arg = labels[label]
    return bytecode_blocks