from typing import Dict, Optional, Tuple
def parse_identity(text):
    name, email = text.rsplit(b'<', 1)
    name = name.strip()
    email = email.rstrip(b'>').strip()
    if not name:
        name = None
    if not email:
        email = None
    return (name, email)