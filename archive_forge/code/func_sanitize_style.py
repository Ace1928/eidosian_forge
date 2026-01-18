import re
from .html import _BaseHTMLProcessor
from .urls import make_safe_absolute_uri
def sanitize_style(self, style):
    style = re.compile('url\\s*\\(\\s*[^\\s)]+?\\s*\\)\\s*').sub(' ', style)
    if not re.match('^([:,;#%.\\sa-zA-Z0-9!]|\\w-\\w|\'[\\s\\w]+\'|"[\\s\\w]+"|\\([\\d,\\s]+\\))*$', style):
        return ''
    if re.sub('\\s*[-\\w]+\\s*:\\s*[^:;]*;?', '', style).strip():
        return ''
    clean = []
    for prop, value in re.findall('([-\\w]+)\\s*:\\s*([^:;]*)', style):
        if not value:
            continue
        if prop.lower() in self.acceptable_css_properties:
            clean.append(prop + ': ' + value + ';')
        elif prop.split('-')[0].lower() in ['background', 'border', 'margin', 'padding']:
            for keyword in value.split():
                if keyword not in self.acceptable_css_keywords and (not self.valid_css_values.match(keyword)):
                    break
            else:
                clean.append(prop + ': ' + value + ';')
        elif self.svgOK and prop.lower() in self.acceptable_svg_properties:
            clean.append(prop + ': ' + value + ';')
    return ' '.join(clean)