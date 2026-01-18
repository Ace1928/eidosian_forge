import keyword as kw
import sympy
from .repr import ReprPrinter
from .str import StrPrinter
def python(expr, **settings):
    """Return Python interpretation of passed expression
    (can be passed to the exec() function without any modifications)"""
    printer = PythonPrinter(settings)
    exprp = printer.doprint(expr)
    result = ''
    renamings = {}
    for symbolname in printer.symbols:
        if '{' in symbolname:
            newsymbolname = symbolname.replace('{', '').replace('}', '')
            renamings[sympy.Symbol(symbolname)] = newsymbolname
        else:
            newsymbolname = symbolname
        if kw.iskeyword(newsymbolname):
            while True:
                newsymbolname += '_'
                if newsymbolname not in printer.symbols and newsymbolname not in printer.functions:
                    renamings[sympy.Symbol(symbolname)] = sympy.Symbol(newsymbolname)
                    break
        result += newsymbolname + " = Symbol('" + symbolname + "')\n"
    for functionname in printer.functions:
        newfunctionname = functionname
        if kw.iskeyword(newfunctionname):
            while True:
                newfunctionname += '_'
                if newfunctionname not in printer.symbols and newfunctionname not in printer.functions:
                    renamings[sympy.Function(functionname)] = sympy.Function(newfunctionname)
                    break
        result += newfunctionname + " = Function('" + functionname + "')\n"
    if renamings:
        exprp = expr.subs(renamings)
    result += 'e = ' + printer._str(exprp)
    return result