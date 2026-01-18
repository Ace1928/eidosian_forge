import re
def parse_siteclass_omegas(line, site_classes):
    """Find omega estimate for each class.

    For models which have multiple site classes, find the omega estimated
    for each class.
    """
    line_floats = re.findall('\\d{1,3}\\.\\d{5}', line)
    if not site_classes or len(line_floats) == 0:
        return
    for n in range(len(line_floats)):
        site_classes[n]['omega'] = line_floats[n]
    return site_classes