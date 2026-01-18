from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_strategy(eg, module):
    persistence = module.params.get('persistence')
    signals = module.params.get('signals')
    eg_strategy = expand_fields(strategy_fields, module.params, 'Strategy')
    terminate_at_end_of_billing_hour = module.params.get('terminate_at_end_of_billing_hour')
    if terminate_at_end_of_billing_hour is not None:
        eg_strategy.eg_scaling_strategy = expand_fields(scaling_strategy_fields, module.params, 'ScalingStrategy')
    if persistence is not None:
        eg_strategy.persistence = expand_fields(persistence_fields, persistence, 'Persistence')
    if signals is not None:
        eg_signals = expand_list(signals, signal_fields, 'Signal')
        if len(eg_signals) > 0:
            eg_strategy.signals = eg_signals
    eg.strategy = eg_strategy