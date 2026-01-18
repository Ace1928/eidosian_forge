from .utils import logger, NO_VALUE
from typing import Mapping, Iterable, Callable, Union, TypeVar, Tuple, Any, List, Set, Optional, Collection, TYPE_CHECKING
def match_examples(self, parse_fn: 'Callable[[str], Tree]', examples: Union[Mapping[T, Iterable[str]], Iterable[Tuple[T, Iterable[str]]]], token_type_match_fallback: bool=False, use_accepts: bool=True) -> Optional[T]:
    """Allows you to detect what's wrong in the input text by matching
        against example errors.

        Given a parser instance and a dictionary mapping some label with
        some malformed syntax examples, it'll return the label for the
        example that bests matches the current error. The function will
        iterate the dictionary until it finds a matching error, and
        return the corresponding value.

        For an example usage, see `examples/error_reporting_lalr.py`

        Parameters:
            parse_fn: parse function (usually ``lark_instance.parse``)
            examples: dictionary of ``{'example_string': value}``.
            use_accepts: Recommended to keep this as ``use_accepts=True``.
        """
    assert self.state is not None, 'Not supported for this exception'
    if isinstance(examples, Mapping):
        examples = examples.items()
    candidate = (None, False)
    for i, (label, example) in enumerate(examples):
        assert not isinstance(example, str), 'Expecting a list'
        for j, malformed in enumerate(example):
            try:
                parse_fn(malformed)
            except UnexpectedInput as ut:
                if ut.state == self.state:
                    if use_accepts and isinstance(self, UnexpectedToken) and isinstance(ut, UnexpectedToken) and (ut.accepts != self.accepts):
                        logger.debug('Different accepts with same state[%d]: %s != %s at example [%s][%s]' % (self.state, self.accepts, ut.accepts, i, j))
                        continue
                    if isinstance(self, (UnexpectedToken, UnexpectedEOF)) and isinstance(ut, (UnexpectedToken, UnexpectedEOF)):
                        if ut.token == self.token:
                            logger.debug('Exact Match at example [%s][%s]' % (i, j))
                            return label
                        if token_type_match_fallback:
                            if ut.token.type == self.token.type and (not candidate[-1]):
                                logger.debug('Token Type Fallback at example [%s][%s]' % (i, j))
                                candidate = (label, True)
                    if candidate[0] is None:
                        logger.debug('Same State match at example [%s][%s]' % (i, j))
                        candidate = (label, False)
    return candidate[0]