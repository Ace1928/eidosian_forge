import pyrfc3339
from ._auth_context import ContextKey
from ._caveat import parse_caveat
from ._conditions import COND_TIME_BEFORE, STD_NAMESPACE
from ._utils import condition_with_prefix
def expiry_time(ns, cavs):
    """ Returns the minimum time of any time-before caveats found
    in the given list or None if no such caveats were found.

    The ns parameter is
    :param ns: used to determine the standard namespace prefix - if
    the standard namespace is not found, the empty prefix is assumed.
    :param cavs: a list of pymacaroons.Caveat
    :return: datetime.DateTime or None.
    """
    prefix = ns.resolve(STD_NAMESPACE)
    time_before_cond = condition_with_prefix(prefix, COND_TIME_BEFORE)
    t = None
    for cav in cavs:
        if not cav.first_party():
            continue
        cav = cav.caveat_id_bytes.decode('utf-8')
        name, rest = parse_caveat(cav)
        if name != time_before_cond:
            continue
        try:
            et = pyrfc3339.parse(rest, utc=True).replace(tzinfo=None)
            if t is None or et < t:
                t = et
        except ValueError:
            continue
    return t