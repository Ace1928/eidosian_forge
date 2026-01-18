from enum import Enum
@staticmethod
def factors_to_string(factors):
    if len(factors) == 0:
        return ''
    parts = []
    factors_list = sorted(factors)
    begin = factors_list[0]
    for i in range(1, len(factors_list)):
        if factors_list[i] != factors_list[i - 1] + 1:
            end = factors_list[i - 1]
            if begin != end:
                parts.append('{}-{}'.format(begin, end))
            else:
                parts.append(str(begin))
            begin = factors_list[i]
    end = len(factors_list) - 1
    if begin != factors_list[end]:
        parts.append('{}-{}'.format(begin, factors_list[end]))
    else:
        parts.append(str(begin))
    return ';'.join(parts)