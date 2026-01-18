from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def show_sql(self, expression: exp.Show) -> str:
    this = f' {expression.name}'
    full = ' FULL' if expression.args.get('full') else ''
    global_ = ' GLOBAL' if expression.args.get('global') else ''
    target = self.sql(expression, 'target')
    target = f' {target}' if target else ''
    if expression.name in ('COLUMNS', 'INDEX'):
        target = f' FROM{target}'
    elif expression.name == 'GRANTS':
        target = f' FOR{target}'
    db = self._prefixed_sql('FROM', expression, 'db')
    like = self._prefixed_sql('LIKE', expression, 'like')
    where = self.sql(expression, 'where')
    types = self.expressions(expression, key='types')
    types = f' {types}' if types else types
    query = self._prefixed_sql('FOR QUERY', expression, 'query')
    if expression.name == 'PROFILE':
        offset = self._prefixed_sql('OFFSET', expression, 'offset')
        limit = self._prefixed_sql('LIMIT', expression, 'limit')
    else:
        offset = ''
        limit = self._oldstyle_limit_sql(expression)
    log = self._prefixed_sql('IN', expression, 'log')
    position = self._prefixed_sql('FROM', expression, 'position')
    channel = self._prefixed_sql('FOR CHANNEL', expression, 'channel')
    if expression.name == 'ENGINE':
        mutex_or_status = ' MUTEX' if expression.args.get('mutex') else ' STATUS'
    else:
        mutex_or_status = ''
    return f'SHOW{full}{global_}{this}{target}{types}{db}{query}{log}{position}{channel}{mutex_or_status}{like}{where}{offset}{limit}'