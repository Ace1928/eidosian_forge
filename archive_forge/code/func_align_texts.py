import math
def align_texts(source_blocks, target_blocks, params=LanguageIndependent):
    """Creates the sentence alignment of two texts.

    Texts can consist of several blocks. Block boundaries cannot be crossed by sentence
    alignment links.

    Each block consists of a list that contains the lengths (in characters) of the sentences
    in this block.

    @param source_blocks: The list of blocks in the source text.
    @param target_blocks: The list of blocks in the target text.
    @param params: the sentence alignment parameters.

    @returns: A list of sentence alignment lists
    """
    if len(source_blocks) != len(target_blocks):
        raise ValueError('Source and target texts do not have the same number of blocks.')
    return [align_blocks(source_block, target_block, params) for source_block, target_block in zip(source_blocks, target_blocks)]