from oslo_log import log as logging
from neutron_lib.api import validators as lib_validators
from neutron_lib.callbacks import events
from neutron_lib.callbacks import registry
from neutron_lib.services.qos import constants
def is_rule_supported(self, rule):
    supported_parameters = self.supported_rules.get(rule.rule_type)
    if not supported_parameters:
        LOG.debug('Rule type %(rule_type)s is not supported by %(driver_name)s', {'rule_type': rule.rule_type, 'driver_name': self.name})
        return False
    for parameter, validators in supported_parameters.items():
        parameter_value = rule.get(parameter)
        for validator_type, validator_data in validators.items():
            validator_function = lib_validators.get_validator(validator_type)
            validate_result = validator_function(parameter_value, validator_data)
            if validate_result:
                LOG.debug('Parameter %(parameter)s=%(value)s in rule type %(rule_type)s is not supported by %(driver_name)s. Validate result: %(validate_result)s', {'parameter': parameter, 'value': parameter_value, 'rule_type': rule.rule_type, 'driver_name': self.name, 'validate_result': validate_result})
                return False
    return True