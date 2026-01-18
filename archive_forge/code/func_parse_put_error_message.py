import re
from lxml import etree
def parse_put_error_message(message):
    error = parse_error_message(message)
    required_fields = []
    if error:
        for line in error.split('\n'):
            try:
                datatype_name = re.findall("'.*?'", line)[0].strip("'")
                element_name = re.findall("'.*?'", line)[1].rsplit(':', 1)[1].strip("}'")
                required_fields.append((datatype_name, element_name))
            except Exception:
                continue
    return required_fields