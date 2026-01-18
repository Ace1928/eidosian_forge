import json
import xml.etree.ElementTree as ET
from langcodes.util import data_filename
from langcodes.registry_parser import parse_registry
def read_cldr_supplemental(dataname):
    cldr_supp_path = data_filename('cldr-json/cldr-json/cldr-core/supplemental')
    filename = data_filename(f'{cldr_supp_path}/{dataname}.json')
    fulldata = json.load(open(filename, encoding='utf-8'))
    if dataname == 'aliases':
        data = fulldata['supplemental']['metadata']['alias']
    else:
        data = fulldata['supplemental'][dataname]
    return data