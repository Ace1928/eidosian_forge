from contextlib import contextmanager
from typing import (
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Context, Leaf, Node, RawNode, convert
from . import grammar, token, tokenize
def addtoken(self, type: int, value: str, context: Context) -> bool:
    """Add a token; return True iff this is the end of the program."""
    ilabels = self.classify(type, value, context)
    assert len(ilabels) >= 1
    if len(ilabels) == 1:
        [ilabel] = ilabels
        return self._addtoken(ilabel, type, value, context)
    with self.proxy.release() as proxy:
        counter, force = (0, False)
        recorder = Recorder(self, ilabels, context)
        recorder.add_token(type, value, raw=True)
        next_token_value = value
        while recorder.determine_route(next_token_value) is None:
            if not proxy.can_advance(counter):
                force = True
                break
            next_token_type, next_token_value, *_ = proxy.eat(counter)
            if next_token_type in (tokenize.COMMENT, tokenize.NL):
                counter += 1
                continue
            if next_token_type == tokenize.OP:
                next_token_type = grammar.opmap[next_token_value]
            recorder.add_token(next_token_type, next_token_value)
            counter += 1
        ilabel = cast(int, recorder.determine_route(next_token_value, force=force))
        assert ilabel is not None
    return self._addtoken(ilabel, type, value, context)