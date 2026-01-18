import os
def get_boolean_option(option_dict, option_name, env_name):
    return option_name in option_dict and option_dict[option_name][1].lower() in TRUE_VALUES or str(os.getenv(env_name)).lower() in TRUE_VALUES