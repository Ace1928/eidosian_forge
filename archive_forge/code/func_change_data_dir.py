from boto.pyami.installers.ubuntu.installer import Installer
import os
import boto
from boto.utils import ShellCommand
from boto.compat import ConfigParser
import time
def change_data_dir(self, password=None):
    data_dir = boto.config.get('MySQL', 'data_dir', '/mnt')
    fresh_install = False
    is_mysql_running_command = ShellCommand('mysqladmin ping')
    is_mysql_running_command.run()
    if is_mysql_running_command.getStatus() == 0:
        time.sleep(10)
        i = 0
        while self.run("echo 'quit' | mysql -u root") != 0 and i < 5:
            time.sleep(5)
            i = i + 1
        self.run('/etc/init.d/mysql stop')
        self.run('pkill -9 mysql')
    mysql_path = os.path.join(data_dir, 'mysql')
    if not os.path.exists(mysql_path):
        self.run('mkdir %s' % mysql_path)
        fresh_install = True
    self.run('chown -R mysql:mysql %s' % mysql_path)
    fp = open('/etc/mysql/conf.d/use_mnt.cnf', 'w')
    fp.write('# created by pyami\n')
    fp.write('# use the %s volume for data\n' % data_dir)
    fp.write('[mysqld]\n')
    fp.write('datadir = %s\n' % mysql_path)
    fp.write('log_bin = %s\n' % os.path.join(mysql_path, 'mysql-bin.log'))
    fp.close()
    if fresh_install:
        self.run('cp -pr /var/lib/mysql/* %s/' % mysql_path)
        self.start('mysql')
    else:
        config_parser = ConfigParser()
        config_parser.read('/etc/mysql/debian.cnf')
        password = config_parser.get('client', 'password')
        self.start('mysql')
        time.sleep(10)
        grant_command = 'echo "GRANT ALL PRIVILEGES ON *.* TO \'debian-sys-maint\'@\'localhost\' IDENTIFIED BY \'%s\' WITH GRANT OPTION;" | mysql' % password
        while self.run(grant_command) != 0:
            time.sleep(5)