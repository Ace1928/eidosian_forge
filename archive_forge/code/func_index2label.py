from typing import Iterator, List, Optional, Tuple, cast
from ._protocols import PdfCommonDocProtocol
from ._utils import logger_warning
from .generic import ArrayObject, DictionaryObject, NullObject, NumberObject
def index2label(reader: PdfCommonDocProtocol, index: int) -> str:
    """
    See 7.9.7 "Number Trees".

    Args:
        reader: The PdfReader
        index: The index of the page

    Returns:
        The label of the page, e.g. "iv" or "4".
    """
    root = cast(DictionaryObject, reader.root_object)
    if '/PageLabels' not in root:
        return str(index + 1)
    number_tree = cast(DictionaryObject, root['/PageLabels'].get_object())
    if '/Nums' in number_tree:
        return get_label_from_nums(number_tree, index)
    if '/Kids' in number_tree and (not isinstance(number_tree['/Kids'], NullObject)):
        level = 0
        while level < 100:
            kids = cast(List[DictionaryObject], number_tree['/Kids'])
            for kid in kids:
                limits = cast(List[int], kid['/Limits'])
                if limits[0] <= index <= limits[1]:
                    if kid.get('/Kids', None) is not None:
                        level += 1
                        if level == 100:
                            raise NotImplementedError('Too deep nesting is not supported.')
                        number_tree = kid
                        break
                    return get_label_from_nums(kid, index)
            else:
                break
    logger_warning(f'Could not reliably determine page label for {index}.', __name__)
    return str(index + 1)