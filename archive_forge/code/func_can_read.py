import paramiko
import paramiko.client
def can_read(self):
    return self.channel.recv_ready()