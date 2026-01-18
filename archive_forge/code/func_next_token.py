import json
from ..error import GraphQLSyntaxError
def next_token(self, reset_position=None):
    if reset_position is None:
        reset_position = self.prev_position
    token = read_token(self.source, reset_position)
    self.prev_position = token.end
    return token