from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import OuterProduct, Operator
from sympy.physics.quantum.state import State, KetBase, BraBase, Wavefunction
from sympy.physics.quantum.tensorproduct import TensorProduct
def qapply_Mul(e, **options):
    ip_doit = options.get('ip_doit', True)
    args = list(e.args)
    if len(args) <= 1 or not isinstance(e, Mul):
        return e
    rhs = args.pop()
    lhs = args.pop()
    if not isinstance(rhs, Wavefunction) and sympify(rhs).is_commutative or (not isinstance(lhs, Wavefunction) and sympify(lhs).is_commutative):
        return e
    if isinstance(lhs, Pow) and lhs.exp.is_Integer:
        args.append(lhs.base ** (lhs.exp - 1))
        lhs = lhs.base
    if isinstance(lhs, OuterProduct):
        args.append(lhs.ket)
        lhs = lhs.bra
    if isinstance(lhs, (Commutator, AntiCommutator)):
        comm = lhs.doit()
        if isinstance(comm, Add):
            return qapply(e.func(*args + [comm.args[0], rhs]) + e.func(*args + [comm.args[1], rhs]), **options)
        else:
            return qapply(e.func(*args) * comm * rhs, **options)
    if isinstance(lhs, TensorProduct) and all((isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in lhs.args)) and isinstance(rhs, TensorProduct) and all((isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in rhs.args)) and (len(lhs.args) == len(rhs.args)):
        result = TensorProduct(*[qapply(lhs.args[n] * rhs.args[n], **options) for n in range(len(lhs.args))]).expand(tensorproduct=True)
        return qapply_Mul(e.func(*args), **options) * result
    try:
        result = lhs._apply_operator(rhs, **options)
    except (NotImplementedError, AttributeError):
        try:
            result = rhs._apply_from_right_to(lhs, **options)
        except (NotImplementedError, AttributeError):
            if isinstance(lhs, BraBase) and isinstance(rhs, KetBase):
                result = InnerProduct(lhs, rhs)
                if ip_doit:
                    result = result.doit()
            else:
                result = None
    if result == 0:
        return S.Zero
    elif result is None:
        if len(args) == 0:
            return e
        else:
            return qapply_Mul(e.func(*args + [lhs]), **options) * rhs
    elif isinstance(result, InnerProduct):
        return result * qapply_Mul(e.func(*args), **options)
    else:
        return qapply(e.func(*args) * result, **options)