from typing import Any, Dict, List, NamedTuple, Optional, Union
from graphql import (
from graphql import GraphQLNamedOutputType
@property
def startCursor(self) -> Optional[ConnectionCursor]:
    ...