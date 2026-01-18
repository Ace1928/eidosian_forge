import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import template as cfn_template
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters
from heat.engine import rsrc_defn
from heat.engine import template_common
def resource_definitions(self, stack):
    resources = self.t.get(self.RESOURCES) or {}
    conditions = self.conditions(stack)
    valid_keys = frozenset(self._RESOURCE_KEYS)

    def defns():
        for name, snippet in resources.items():
            try:
                invalid_keys = set(snippet) - valid_keys
                if invalid_keys:
                    raise ValueError(_('Invalid keyword(s) inside a resource definition: %s') % ', '.join(invalid_keys))
                defn_data = dict(self._rsrc_defn_args(stack, name, snippet))
            except (TypeError, ValueError, KeyError) as ex:
                msg = str(ex)
                raise exception.StackValidationFailed(message=msg)
            defn = rsrc_defn.ResourceDefinition(name, **defn_data)
            cond_name = defn.condition()
            if cond_name is not None:
                try:
                    enabled = conditions.is_enabled(cond_name)
                except ValueError as exc:
                    path = [self.RESOURCES, name, self.RES_CONDITION]
                    message = str(exc)
                    raise exception.StackValidationFailed(path=path, message=message)
                if not enabled:
                    continue
            yield (name, defn)
    return dict(defns())