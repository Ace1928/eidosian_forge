from typing import Any, Dict, Optional
def assert_equivalent_repr(value: Any, *, setup_code: str='import cirq\nimport numpy as np\nimport sympy\nimport pandas as pd\nimport datetime\n', global_vals: Optional[Dict[str, Any]]=None, local_vals: Optional[Dict[str, Any]]=None) -> None:
    """Checks that eval(repr(v)) == v.

    Args:
        value: A value whose repr should be evaluatable python
            code that produces an equivalent value.
        setup_code: Code that must be executed before the repr can be evaluated.
            Ideally this should just be a series of 'import' lines.
        global_vals: Pre-defined values that should be in the global scope when
            evaluating the repr.
        local_vals: Pre-defined values that should be in the local scope when
            evaluating the repr.

    Raises:
        AssertionError: If the assertion fails, or eval(repr(value)) raises an error.
    """
    __tracebackhide__ = True
    global_vals = global_vals or {}
    local_vals = local_vals or {}
    exec(setup_code, global_vals, local_vals)
    try:
        eval_repr_value = eval(repr(value), global_vals, local_vals)
    except Exception as ex:
        raise AssertionError(f'eval(repr(value)) raised an exception.\n\nsetup_code={setup_code}\ntype(value): {type(value)}\nvalue={value!r}\nerror={ex!r}')
    assert eval_repr_value == value, "The repr of a value of type {} didn't evaluate to something equal to the value.\neval(repr(value)) != value\n\nvalue: {}\nrepr(value): {!r}\neval(repr(value)): {}\nrepr(eval(repr(value))): {!r}\n\ntype(value): {}\ntype(eval(repr(value))): {!r}\n\nsetup_code:\n{}\n".format(type(value), value, repr(value), eval_repr_value, repr(eval_repr_value), type(value), type(eval_repr_value), '    ' + setup_code.replace('\n', '\n    '))
    try:
        a = eval(f'{value!r}.__class__', global_vals, local_vals)
    except Exception:
        raise AssertionError(f"The repr of a value of type {type(value)} wasn't 'dottable'.\n{value!r}.XXX must be equivalent to ({value!r}).XXX, but it raised an error instead.")
    b = eval(f'({value!r}).__class__', global_vals, local_vals)
    assert a == b, f"The repr of a value of type {type(value)} wasn't 'dottable'.\n{value!r}.XXX must be equivalent to ({value!r}).XXX, but it wasn't."