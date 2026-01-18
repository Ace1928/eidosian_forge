from __future__ import absolute_import, division, print_function
import re
from .common import F5ModuleError
def parseStats(entry):
    if 'description' in entry:
        return entry['description']
    elif 'value' in entry:
        return entry['value']
    elif 'entries' in entry or ('nestedStats' in entry and 'entries' in entry['nestedStats']):
        if 'entries' in entry:
            entries = entry['entries']
        else:
            entries = entry['nestedStats']['entries']
        result = None
        for name in entries:
            entry = entries[name]
            if 'https://localhost' in name:
                name = name.split('/')
                name = name[-1]
                if result and isinstance(result, list):
                    result.append(parseStats(entry))
                elif result and isinstance(result, dict):
                    result[name] = parseStats(entry)
                else:
                    try:
                        int(name)
                        result = list()
                        result.append(parseStats(entry))
                    except ValueError:
                        result = dict()
                        result[name] = parseStats(entry)
            elif '.' in name:
                names = name.split('.')
                key = names[0]
                value = names[1]
                if result is None:
                    result = dict()
                    result[key] = dict()
                elif key not in result:
                    result[key] = dict()
                elif result[key] is None:
                    result[key] = dict()
                result[key][value] = parseStats(entry)
            elif result and isinstance(result, list):
                result.append(parseStats(entry))
            elif result and isinstance(result, dict):
                result[name] = parseStats(entry)
            else:
                try:
                    int(name)
                    result = list()
                    result.append(parseStats(entry))
                except ValueError:
                    result = dict()
                    result[name] = parseStats(entry)
        return result