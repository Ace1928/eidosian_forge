import re
from kombu.utils.encoding import safe_str
def parse_search_terms(raw_search_value):
    search_regexp = '(?:[^\\s,"]|"(?:\\\\.|[^"])*")+'
    if not raw_search_value:
        return {}
    parsed_search = {}
    for query_part in re.findall(search_regexp, raw_search_value):
        if not query_part:
            continue
        if query_part.startswith('result:'):
            parsed_search['result'] = preprocess_search_value(query_part[len('result:'):])
        elif query_part.startswith('args:'):
            if 'args' not in parsed_search:
                parsed_search['args'] = []
            parsed_search['args'].append(preprocess_search_value(query_part[len('args:'):]))
        elif query_part.startswith('kwargs:'):
            if 'kwargs' not in parsed_search:
                parsed_search['kwargs'] = {}
            try:
                key, value = [p.strip() for p in query_part[len('kwargs:'):].split('=')]
            except ValueError:
                continue
            parsed_search['kwargs'][key] = preprocess_search_value(value)
        elif query_part.startswith('state'):
            if 'state' not in parsed_search:
                parsed_search['state'] = []
            parsed_search['state'].append(preprocess_search_value(query_part[len('state:'):]))
        else:
            parsed_search['any'] = preprocess_search_value(query_part)
    return parsed_search