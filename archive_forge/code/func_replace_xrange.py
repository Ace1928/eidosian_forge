import logging
import re
def replace_xrange(comp, loose):
    comp = comp.strip()
    r = regexp[XRANGELOOSE] if loose else regexp[XRANGE]

    def repl(mob):
        ret = mob.group(0)
        gtlt, M, m, p, pr, _ = mob.groups()
        logger.debug('xrange %s %s %s %s %s %s %s', comp, ret, gtlt, M, m, p, pr)
        xM = is_x(M)
        xm = xM or is_x(m)
        xp = xm or is_x(p)
        any_x = xp
        if gtlt == '=' and any_x:
            gtlt = ''
        logger.debug('xrange gtlt=%s any_x=%s', gtlt, any_x)
        if xM:
            if gtlt == '>' or gtlt == '<':
                ret = '<0.0.0'
            else:
                ret = '*'
        elif gtlt and any_x:
            if xm:
                m = 0
            if xp:
                p = 0
            if gtlt == '>':
                gtlt = '>='
                if xm:
                    M = int(M) + 1
                    m = 0
                    p = 0
                elif xp:
                    m = int(m) + 1
                    p = 0
            elif gtlt == '<=':
                gtlt = '<'
                if xm:
                    M = int(M) + 1
                else:
                    m = int(m) + 1
            ret = gtlt + str(M) + '.' + str(m) + '.' + str(p)
        elif xm:
            ret = '>=' + M + '.0.0 <' + str(int(M) + 1) + '.0.0'
        elif xp:
            ret = '>=' + M + '.' + m + '.0 <' + M + '.' + str(int(m) + 1) + '.0'
        logger.debug('xRange return %s', ret)
        return ret
    return r.sub(repl, comp)