from ...attrs import LIKE_NUM
def suffix_filter(text):
    for num_suffix in _numeral_suffixes.keys():
        length = len(num_suffix)
        if len(text) < length:
            break
        elif text.endswith(num_suffix):
            return text[:-length] + _numeral_suffixes[num_suffix]
    return text