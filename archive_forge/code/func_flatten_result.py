import re
from django.utils.functional import SimpleLazyObject
def flatten_result(source):
    """
    Turn the given source sequence into a list of reg-exp possibilities and
    their arguments. Return a list of strings and a list of argument lists.
    Each of the two lists will be of the same length.
    """
    if source is None:
        return ([''], [[]])
    if isinstance(source, Group):
        if source[1] is None:
            params = []
        else:
            params = [source[1]]
        return ([source[0]], [params])
    result = ['']
    result_args = [[]]
    pos = last = 0
    for pos, elt in enumerate(source):
        if isinstance(elt, str):
            continue
        piece = ''.join(source[last:pos])
        if isinstance(elt, Group):
            piece += elt[0]
            param = elt[1]
        else:
            param = None
        last = pos + 1
        for i in range(len(result)):
            result[i] += piece
            if param:
                result_args[i].append(param)
        if isinstance(elt, (Choice, NonCapture)):
            if isinstance(elt, NonCapture):
                elt = [elt]
            inner_result, inner_args = ([], [])
            for item in elt:
                res, args = flatten_result(item)
                inner_result.extend(res)
                inner_args.extend(args)
            new_result = []
            new_args = []
            for item, args in zip(result, result_args):
                for i_item, i_args in zip(inner_result, inner_args):
                    new_result.append(item + i_item)
                    new_args.append(args[:] + i_args)
            result = new_result
            result_args = new_args
    if pos >= last:
        piece = ''.join(source[last:])
        for i in range(len(result)):
            result[i] += piece
    return (result, result_args)