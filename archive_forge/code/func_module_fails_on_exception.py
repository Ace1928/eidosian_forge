from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible_collections.community.general.plugins.module_utils.mh.exceptions import ModuleHelperException
def module_fails_on_exception(func):
    conflict_list = ('msg', 'exception', 'output', 'vars', 'changed')

    @wraps(func)
    def wrapper(self, *args, **kwargs):

        def fix_var_conflicts(output):
            result = dict([(k if k not in conflict_list else '_' + k, v) for k, v in output.items()])
            return result
        try:
            func(self, *args, **kwargs)
        except SystemExit:
            raise
        except ModuleHelperException as e:
            if e.update_output:
                self.update_output(e.update_output)
            output = fix_var_conflicts(self.output)
            self.module.fail_json(msg=e.msg, exception=traceback.format_exc(), output=self.output, vars=self.vars.output(), **output)
        except Exception as e:
            output = fix_var_conflicts(self.output)
            msg = 'Module failed with exception: {0}'.format(str(e).strip())
            self.module.fail_json(msg=msg, exception=traceback.format_exc(), output=self.output, vars=self.vars.output(), **output)
    return wrapper