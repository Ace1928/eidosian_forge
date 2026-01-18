import logging
import re
def replace_stars(comp, loose):
    logger.debug('replaceStars %s %s', comp, loose)
    return regexp[STAR].sub('', comp.strip())