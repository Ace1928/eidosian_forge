import logging
import re
def replace_xranges(comp, loose):
    logger.debug('replaceXRanges %s %s', comp, loose)
    return ' '.join([replace_xrange(c, loose) for c in re.split('\\s+', comp.strip())])