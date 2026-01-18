import random
import string
def random_zone_file(name='testzoneimport'):
    return '$ORIGIN {0}{1}.com.\n$TTL 300\n{0}{1}.com. 300 IN SOA ns.{0}{1}.com. nsadmin.{0}{1}.com. 42 42 42 42 42\n{0}{1}.com. 300 IN NS ns.{0}{1}.com.\n{0}{1}.com. 300 IN MX 10 mail.{0}{1}.com.\nns.{0}{1}.com. 300 IN A 10.0.0.1\nmail.{0}{1}.com. 300 IN A 10.0.0.2\n'.format(name, random_digits())