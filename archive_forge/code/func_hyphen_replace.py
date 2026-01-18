import logging
import re
def hyphen_replace(mob):
    from_, fM, fm, fp, fpr, fb, to, tM, tm, tp, tpr, tb = mob.groups()
    if is_x(fM):
        from_ = ''
    elif is_x(fm):
        from_ = '>=' + fM + '.0.0'
    elif is_x(fp):
        from_ = '>=' + fM + '.' + fm + '.0'
    else:
        from_ = '>=' + from_
    if is_x(tM):
        to = ''
    elif is_x(tm):
        to = '<' + str(int(tM) + 1) + '.0.0'
    elif is_x(tp):
        to = '<' + tM + '.' + str(int(tm) + 1) + '.0'
    elif tpr:
        to = '<=' + tM + '.' + tm + '.' + tp + '-' + tpr
    else:
        to = '<=' + to
    return (from_ + ' ' + to).strip()