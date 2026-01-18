from jmespath.compat import with_str_method
def set_expression(self, expression):
    self.expression = expression
    self.lex_position = len(expression)
    self.token_type = None
    self.token_value = None