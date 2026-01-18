import re
from pygments.lexer import RegexLexer, include
from pygments.util import get_bool_opt, get_list_opt
from pygments.token import Text, Comment, Operator, Keyword, Name, \
def set_dialect(self, dialect_id):
    if dialect_id not in self.dialects:
        dialect = 'unknown'
    else:
        dialect = dialect_id
    lexemes_to_reject_set = set()
    for list in self.lexemes_to_reject_db[dialect]:
        lexemes_to_reject_set.update(set(list))
    reswords_set = set()
    for list in self.reserved_words_db[dialect]:
        reswords_set.update(set(list))
    builtins_set = set()
    for list in self.builtins_db[dialect]:
        builtins_set.update(set(list).difference(reswords_set))
    pseudo_builtins_set = set()
    for list in self.pseudo_builtins_db[dialect]:
        pseudo_builtins_set.update(set(list).difference(reswords_set))
    adts_set = set()
    for list in self.stdlib_adts_db[dialect]:
        adts_set.update(set(list).difference(reswords_set))
    modules_set = set()
    for list in self.stdlib_modules_db[dialect]:
        modules_set.update(set(list).difference(builtins_set))
    types_set = set()
    for list in self.stdlib_types_db[dialect]:
        types_set.update(set(list).difference(builtins_set))
    procedures_set = set()
    for list in self.stdlib_procedures_db[dialect]:
        procedures_set.update(set(list).difference(builtins_set))
    variables_set = set()
    for list in self.stdlib_variables_db[dialect]:
        variables_set.update(set(list).difference(builtins_set))
    constants_set = set()
    for list in self.stdlib_constants_db[dialect]:
        constants_set.update(set(list).difference(builtins_set))
    self.dialect = dialect
    self.lexemes_to_reject = lexemes_to_reject_set
    self.reserved_words = reswords_set
    self.builtins = builtins_set
    self.pseudo_builtins = pseudo_builtins_set
    self.adts = adts_set
    self.modules = modules_set
    self.types = types_set
    self.procedures = procedures_set
    self.variables = variables_set
    self.constants = constants_set