import random
import string
def random_a_recordset_name(zone_name, recordset_name='testrecord'):
    return f'{recordset_name}{random_digits()}.{zone_name}'