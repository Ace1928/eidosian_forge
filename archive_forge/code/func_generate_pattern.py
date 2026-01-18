import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_pattern(self):
    with self.l('if isinstance({variable}, str):'):
        pattern = self._definition['pattern']
        safe_pattern = pattern.replace('\\', '\\\\').replace('"', '\\"')
        end_of_string_fixed_pattern = DOLLAR_FINDER.sub('\\\\Z', pattern)
        self._compile_regexps[pattern] = re.compile(end_of_string_fixed_pattern)
        with self.l('if not REGEX_PATTERNS[{}].search({variable}):', repr(pattern)):
            self.exc('{name} must match pattern {}', safe_pattern, rule='pattern')