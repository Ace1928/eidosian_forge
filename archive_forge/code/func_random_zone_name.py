import random
import string
def random_zone_name(name='testdomain', tld='com'):
    return f'{name}{random_digits()}.{tld}.'