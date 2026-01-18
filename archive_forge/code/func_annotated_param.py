import copy
import inspect
import re
from typing import (
from ..exceptions import InvalidOperationError
from ..utils.assertion import assert_or_throw
from ..utils.convert import get_full_type_path
from ..utils.entry_points import load_entry_point
from ..utils.hash import to_uuid
from .dict import IndexedOrderedDict
@no_type_check
@classmethod
def annotated_param(cls, annotation: Any, code: Optional[str]=None, matcher: Optional[Callable[[Any], bool]]=None, child_can_reuse_code: bool=False):
    """The decorator to register a type annotation for this function
        wrapper

        :param annotation: the type annotation
        :param code: the single char code to represent this type annotation
            , defaults to None, meaning it will try to use its parent class'
            code, this is allowed only if ``child_can_reuse_code`` is set to
            True on the parent class.
        :param matcher: a function taking in a type annotation and decide
            whether it is acceptable by the :class:`~.AnnotatedParam`
            , defaults to None, meaning it will just do a simple ``==`` check.
        :param child_can_reuse_code: whether the derived types of the current
            AnnotatedParam can reuse the code (if not specifying a new code)
            , defaults to False
        """

    def _func(tp: Type['AnnotatedParam']) -> Type['AnnotatedParam']:
        if not issubclass(tp, AnnotatedParam):
            raise InvalidOperationError(f'{tp} is not a subclass of AnnotatedParam')
        if matcher is not None:
            _matcher = matcher
        else:
            anno = annotation

            def _m(a: Any) -> bool:
                return a == anno
            _matcher = _m
        tp._annotation = annotation
        if code is not None:
            tp._code = code
        else:
            tp._code = tp.__bases__[0]._code
        if tp._code in cls._REGISTERED_CODES:
            _allow_tp = cls._REGISTERED_CODES[tp._code]
            if _allow_tp is not None and inspect.isclass(tp) and issubclass(tp, _allow_tp):
                pass
            else:
                for _ptp, _a, _c, _ in cls._REGISTERED:
                    if _c == tp._code:
                        if str(_ptp) != str(tp):
                            raise InvalidOperationError(f"param code {_c} is already registered by {_ptp} {_a} so can't be used by {tp} {annotation}")
        elif child_can_reuse_code and inspect.isclass(tp):
            cls._REGISTERED_CODES[tp._code] = tp
        else:
            cls._REGISTERED_CODES[tp._code] = None
        cls._REGISTERED.append((tp, annotation, code, _matcher))
        return tp
    return _func