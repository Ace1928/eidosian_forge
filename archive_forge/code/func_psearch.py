import gc
import re
import sys
from IPython.core import page
from IPython.core.error import StdinNotImplementedError, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.encoding import DEFAULT_ENCODING
from IPython.utils.openpy import read_py_file
from IPython.utils.path import get_py_filename
@line_magic
def psearch(self, parameter_s=''):
    """Search for object in namespaces by wildcard.

        %psearch [options] PATTERN [OBJECT TYPE]

        Note: ? can be used as a synonym for %psearch, at the beginning or at
        the end: both a*? and ?a* are equivalent to '%psearch a*'.  Still, the
        rest of the command line must be unchanged (options come first), so
        for example the following forms are equivalent

        %psearch -i a* function
        -i a* function?
        ?-i a* function

        Arguments:

          PATTERN

          where PATTERN is a string containing * as a wildcard similar to its
          use in a shell.  The pattern is matched in all namespaces on the
          search path. By default objects starting with a single _ are not
          matched, many IPython generated objects have a single
          underscore. The default is case insensitive matching. Matching is
          also done on the attributes of objects and not only on the objects
          in a module.

          [OBJECT TYPE]

          Is the name of a python type from the types module. The name is
          given in lowercase without the ending type, ex. StringType is
          written string. By adding a type here only objects matching the
          given type are matched. Using all here makes the pattern match all
          types (this is the default).

        Options:

          -a: makes the pattern match even objects whose names start with a
          single underscore.  These names are normally omitted from the
          search.

          -i/-c: make the pattern case insensitive/sensitive.  If neither of
          these options are given, the default is read from your configuration
          file, with the option ``InteractiveShell.wildcards_case_sensitive``.
          If this option is not specified in your configuration file, IPython's
          internal default is to do a case sensitive search.

          -e/-s NAMESPACE: exclude/search a given namespace.  The pattern you
          specify can be searched in any of the following namespaces:
          'builtin', 'user', 'user_global','internal', 'alias', where
          'builtin' and 'user' are the search defaults.  Note that you should
          not use quotes when specifying namespaces.

          -l: List all available object types for object matching. This function
          can be used without arguments.

          'Builtin' contains the python module builtin, 'user' contains all
          user data, 'alias' only contain the shell aliases and no python
          objects, 'internal' contains objects used by IPython.  The
          'user_global' namespace is only used by embedded IPython instances,
          and it contains module-level globals.  You can add namespaces to the
          search with -s or exclude them with -e (these options can be given
          more than once).

        Examples
        --------
        ::

          %psearch a*            -> objects beginning with an a
          %psearch -e builtin a* -> objects NOT in the builtin space starting in a
          %psearch a* function   -> all functions beginning with an a
          %psearch re.e*         -> objects beginning with an e in module re
          %psearch r*.e*         -> objects that start with e in modules starting in r
          %psearch r*.* string   -> all strings in modules beginning with r

        Case sensitive search::

          %psearch -c a*         list all object beginning with lower case a

        Show objects beginning with a single _::

          %psearch -a _*         list objects beginning with a single underscore

        List available objects::

          %psearch -l            list all available object types
        """
    def_search = ['user_local', 'user_global', 'builtin']
    opts, args = self.parse_options(parameter_s, 'cias:e:l', list_all=True)
    opt = opts.get
    shell = self.shell
    psearch = shell.inspector.psearch
    list_types = False
    if 'l' in opts:
        list_types = True
    if 'i' in opts:
        ignore_case = True
    elif 'c' in opts:
        ignore_case = False
    else:
        ignore_case = not shell.wildcards_case_sensitive
    def_search.extend(opt('s', []))
    ns_exclude = ns_exclude = opt('e', [])
    ns_search = [nm for nm in def_search if nm not in ns_exclude]
    try:
        psearch(args, shell.ns_table, ns_search, show_all=opt('a'), ignore_case=ignore_case, list_types=list_types)
    except:
        shell.showtraceback()