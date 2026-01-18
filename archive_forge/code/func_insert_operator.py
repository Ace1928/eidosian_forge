import collections
import re
import uuid
from ply import lex
from ply import yacc
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import lexer
from yaql.language import parser
from yaql.language import utils
def insert_operator(self, existing_operator, existing_operator_binary, new_operator, new_operator_type, create_group, new_operator_alias=None):
    binary_types = (OperatorType.BINARY_RIGHT_ASSOCIATIVE, OperatorType.BINARY_LEFT_ASSOCIATIVE)
    unary_types = (OperatorType.PREFIX_UNARY, OperatorType.SUFFIX_UNARY)
    position = 0
    if existing_operator is not None:
        position = -1
        for i, t in enumerate(self.operators):
            if len(t) < 2 or t[0] != existing_operator:
                continue
            if existing_operator_binary and t[1] not in binary_types:
                continue
            if not existing_operator_binary and t[1] not in unary_types:
                continue
            position = i
            break
        if position < 0:
            raise ValueError('Operator {0} is not found'.format(existing_operator))
        while position < len(self.operators) and len(self.operators[position]) > 1:
            position += 1
    if create_group:
        if position == len(self.operators):
            self.operators.append(())
            position += 1
        else:
            while position < len(self.operators) and len(self.operators[position]) < 2:
                position += 1
            self.operators.insert(position, ())
    self.operators.insert(position, (new_operator, new_operator_type, new_operator_alias))