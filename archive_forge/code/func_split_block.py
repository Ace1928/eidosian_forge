import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.concrete import ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.flags import CompilerFlags
from _pydevd_frame_eval.vendored.bytecode.instr import Label, SetLineno, Instr
def split_block(self, block, index):
    if not isinstance(block, BasicBlock):
        raise TypeError('expected block')
    block_index = self.get_block_index(block)
    if index < 0:
        raise ValueError('index must be positive')
    block = self._blocks[block_index]
    if index == 0:
        return block
    if index > len(block):
        raise ValueError('index out of the block')
    instructions = block[index:]
    if not instructions:
        if block_index + 1 < len(self):
            return self[block_index + 1]
    del block[index:]
    block2 = BasicBlock(instructions)
    block.next_block = block2
    for block in self[block_index + 1:]:
        self._block_index[id(block)] += 1
    self._blocks.insert(block_index + 1, block2)
    self._block_index[id(block2)] = block_index + 1
    return block2