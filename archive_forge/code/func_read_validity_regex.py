import json
import xml.etree.ElementTree as ET
from langcodes.util import data_filename
from langcodes.registry_parser import parse_registry
def read_validity_regex():
    validity_options = []
    for codetype in ('language', 'region', 'script', 'variant'):
        validity_path = data_filename(f'cldr/common/validity/{codetype}.xml')
        root = ET.fromstring(open(validity_path).read())
        matches = root.findall('./idValidity/id')
        for match in matches:
            for item in match.text.strip().split():
                if '~' in item:
                    assert item[-2] == '~'
                    prefix = item[:-3]
                    range_start = item[-3]
                    range_end = item[-1]
                    option = f'{prefix}[{range_start}-{range_end}]'
                    validity_options.append(option)
                else:
                    validity_options.append(item)
    options = '|'.join(validity_options)
    return f'^({options})$'