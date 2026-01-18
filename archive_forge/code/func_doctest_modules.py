import doctest
import re
import types
def doctest_modules(modules, verbose=False, print_info=True, extraglobs=dict()):
    finder = doctest.DocTestFinder(parser=DocTestParser())
    full_extraglobals = globs.copy()
    full_extraglobals.update(extraglobs)
    failed, attempted = (0, 0)
    for module in modules:
        if isinstance(module, types.ModuleType):
            runner = doctest.DocTestRunner(verbose=verbose)
            for test in finder.find(module, extraglobs=full_extraglobals):
                runner.run(test)
            result = runner.summarize()
        else:
            result = module(verbose=verbose)
        failed += result.failed
        attempted += result.attempted
        if print_info:
            print_results(module, result)
    if print_info:
        print('\nAll doctests:\n   %s failures out of %s tests.' % (failed, attempted))
    return doctest.TestResults(failed, attempted)