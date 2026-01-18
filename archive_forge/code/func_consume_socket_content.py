import threading
import socket
import select
def consume_socket_content(sock, timeout=0.5):
    chunks = 65536
    content = b''
    while True:
        more_to_read = select.select([sock], [], [], timeout)[0]
        if not more_to_read:
            break
        new_content = sock.recv(chunks)
        if not new_content:
            break
        content += new_content
    return content