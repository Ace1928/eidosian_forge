import sys
def post_parse(self):
    """Set ``start``, ``join_str`` and ``safe`` attributes.

        After parsing the variable, we need to set up these attributes and it
        only makes sense to do it in a more easily testable way.
        """
    self.safe = ''
    self.start = self.join_str = self.operator
    if self.operator == '+':
        self.start = ''
    if self.operator in ('+', '#', ''):
        self.join_str = ','
    if self.operator == '#':
        self.start = '#'
    if self.operator == '?':
        self.start = '?'
        self.join_str = '&'
    if self.operator in ('+', '#'):
        self.safe = URIVariable.reserved