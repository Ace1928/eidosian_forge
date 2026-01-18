import paramiko
import paramiko.client
@property
def stderr(self):
    return self.channel.makefile_stderr('rb')